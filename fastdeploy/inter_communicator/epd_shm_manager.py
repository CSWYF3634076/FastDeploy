"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import annotations

import time
import uuid
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Optional

import numpy as np

from fastdeploy.utils import llm_logger


def _sanitize_name(name: str, max_len: int = 48) -> str:
    filtered = "".join(ch if ch.isalnum() else "_" for ch in name)
    return filtered[:max_len] if len(filtered) > max_len else filtered


class EPDShmManager:
    """
    Lightweight SHM manager used by EPD.

    It stores numpy payloads in shared memory and returns metadata refs that can
    be serialized into requests and restored by another process.
    """

    def __init__(
        self,
        shm_dir: str = "/dev/shm",
        ttl_sec: int = 120,
        max_bytes: int = 4 * 1024**3,
        name_prefix: str = "fd_epd",
    ) -> None:
        self.shm_dir = shm_dir
        self.ttl_sec = ttl_sec
        self.max_bytes = max_bytes
        self.name_prefix = name_prefix
        self.records: Dict[str, dict] = {}
        self._last_gc_ts = 0.0
        self._gc_interval_sec = 5.0
        llm_logger.info(
            "[EPD][SHM] init manager, shm_dir=%s ttl_sec=%s max_bytes=%s prefix=%s",
            self.shm_dir,
            self.ttl_sec,
            self.max_bytes,
            self.name_prefix,
        )

    def _gen_name(self, request_id: str, chunk_id: int = 0) -> str:
        req = _sanitize_name(request_id or "req")
        token = uuid.uuid4().hex[:8]
        return f"{self.name_prefix}_{req}_{chunk_id}_{token}"

    def cleanup_expired(self, force: bool = False) -> None:
        """
        Cleanup expired shared memory blocks based on TTL.

        Args:
            force: If True, ignore the GC interval and force cleanup immediately.
        """
        now = time.time()
        if not force and now - self._last_gc_ts < self._gc_interval_sec:
            return
        self._last_gc_ts = now

        expired_names = [
            name for name, info in self.records.items() if now - float(info.get("created_ts", now)) > self.ttl_sec
        ]
        for name in expired_names:
            self._release_by_name(name)
            self.records.pop(name, None)

    def dump_numpy(self, request_id: str, array: np.ndarray, chunk_id: int = 0) -> dict:
        """
        Dump a numpy array to shared memory.

        Args:
            request_id: Request identifier for tracking
            array: Numpy array to dump
            chunk_id: Chunk identifier for multi-chunk payloads

        Returns:
            Dictionary containing metadata for loading the array

        Raises:
            TypeError: If array is not a numpy.ndarray
            ValueError: If array is too large
        """
        self.cleanup_expired()

        if not isinstance(array, np.ndarray):
            raise TypeError(f"EPD dump_numpy only accepts numpy.ndarray, but got {type(array)}")
        if array.nbytes > self.max_bytes:
            raise ValueError(f"EPD payload too large: {array.nbytes} bytes > epd_shm_max_bytes {self.max_bytes} bytes")

        shm_name = self._gen_name(request_id=request_id, chunk_id=chunk_id)
        shm = SharedMemory(name=shm_name, create=True, size=array.nbytes)
        try:
            shm_view = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
            shm_view[...] = array
        finally:
            shm.close()

        created_ts = time.time()
        self.records[shm_name] = {"created_ts": created_ts, "nbytes": array.nbytes}
        llm_logger.info(
            "[EPD][SHM] dump request_id=%s chunk_id=%s shm_name=%s nbytes=%s shape=%s dtype=%s",
            request_id,
            chunk_id,
            shm_name,
            array.nbytes,
            tuple(array.shape),
            str(array.dtype),
        )
        return {
            "name": shm_name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
            "chunk_id": int(chunk_id),
            "request_id": request_id,
            "created_ts": created_ts,
        }

    def load_numpy(self, ref: dict) -> Optional[np.ndarray]:
        """
        Load a numpy array from shared memory using metadata reference.

        Args:
            ref: Metadata dictionary returned by dump_numpy

        Returns:
            Loaded numpy array, or None if the shared memory block is not found

        Raises:
            TypeError: If ref is not a dict
            ValueError: If ref is missing required fields
        """
        self.cleanup_expired()
        if not isinstance(ref, dict):
            raise TypeError(f"EPD load_numpy expects dict ref, but got {type(ref)}")

        shm_name = ref.get("name")
        shape = tuple(ref.get("shape", []))
        dtype = np.dtype(ref.get("dtype"))
        if not shm_name:
            raise ValueError(f"Invalid EPD shm ref without name: {ref}")
        if not shape:
            raise ValueError(f"Invalid EPD shm ref without shape: {ref}")

        try:
            shm = SharedMemory(name=shm_name, create=False)
        except FileNotFoundError:
            llm_logger.error("[EPD][SHM] shm not found while loading: name=%s ref=%s", shm_name, ref)
            return None

        try:
            shm_view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            payload = np.array(shm_view, copy=True)
        finally:
            shm.close()
        llm_logger.info(
            "[EPD][SHM] load request_id=%s chunk_id=%s shm_name=%s shape=%s dtype=%s",
            ref.get("request_id"),
            ref.get("chunk_id"),
            shm_name,
            shape,
            str(dtype),
        )
        return payload

    def _release_by_name(self, shm_name: str) -> None:
        try:
            shm = SharedMemory(name=shm_name, create=False)
            shm.close()
            shm.unlink()
            llm_logger.info("[EPD][SHM] release shm_name=%s", shm_name)
        except FileNotFoundError:
            pass
        except Exception as e:
            llm_logger.warning("[EPD][SHM] release failed shm_name=%s err=%s", shm_name, e)

    def release_ref(self, ref: dict) -> None:
        """
        Release a single shared memory block by its metadata reference.

        Args:
            ref: Metadata dictionary returned by dump_numpy
        """
        if not isinstance(ref, dict):
            return
        shm_name = ref.get("name")
        if not shm_name:
            return
        self._release_by_name(shm_name)
        self.records.pop(shm_name, None)

    def release_refs(self, refs: list[dict]) -> None:
        if not refs:
            return
        for ref in refs:
            self.release_ref(ref)
