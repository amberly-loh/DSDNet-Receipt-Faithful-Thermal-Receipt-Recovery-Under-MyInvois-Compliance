import os
import time
import pytest
from dsdnet_receipt.inference_worker import IsolatedReceiptWorker


def worker_probe(payload):
    if payload=='crash':
        os._exit(19)
    if payload=='invalid':
        raise ValueError('Unreadable image')
    if payload=='hang':
        time.sleep(10)
    return {'value':payload,'pid':os.getpid()}


def test_native_crash_is_contained_and_next_receipt_gets_fresh_worker():
    worker=IsolatedReceiptWorker(worker_probe,timeout=15)
    try:
        first=worker.run('one')
        with pytest.raises(ValueError,match='worker stopped'):
            worker.run('crash')
        second=worker.run('two')
        assert first['pid']!=os.getpid()
        assert second['pid']!=first['pid']
        assert second['value']=='two'
    finally:
        worker.close()


def test_bad_image_does_not_interrupt_following_batch_item():
    worker=IsolatedReceiptWorker(worker_probe,timeout=15)
    try:
        with pytest.raises(ValueError,match='Unreadable image'):
            worker.run('invalid')
        assert worker.run('next')['value']=='next'
    finally:
        worker.close()


def test_concurrent_scan_returns_busy_instead_of_blocking_http_threads():
    worker=IsolatedReceiptWorker(worker_probe,timeout=15)
    worker.lock.acquire()
    try:
        with pytest.raises(ValueError,match='Another receipt is processing'):
            worker.run('competing scan')
        assert worker.process is None
    finally:
        worker.lock.release()
        worker.close()


def test_timed_out_worker_is_stopped_and_replaced():
    worker=IsolatedReceiptWorker(worker_probe,timeout=15)
    try:
        worker.run('ready')
        worker.timeout=.3
        with pytest.raises(ValueError,match='timed out'):
            worker.run('hang')
        worker.timeout=15
        assert worker.run('after')['value']=='after'
    finally:
        worker.close()
