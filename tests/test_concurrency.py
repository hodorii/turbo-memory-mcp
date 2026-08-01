import sqlite3
import threading
import time
import pytest
from concurrency import ConnectionPool, TransactionManager, RetryPolicy, DatabaseLockedError

def test_connection_pool():
    db_path = ":memory:"
    pool = ConnectionPool(db_path, pool_size=2)
    
    # Test 1: Basic acquire/release
    conn1 = pool.acquire()
    assert isinstance(conn1, sqlite3.Connection)
    pool.release(conn1)
    
    # Test 2: Pool size limit
    conn1 = pool.acquire()
    conn2 = pool.acquire()
    
    def acquire_3():
        conn3 = pool.acquire()
        pool.release(conn3)
        
    t = threading.Thread(target=acquire_3)
    t.start()
    time.sleep(0.1)
    pool.release(conn1)
    t.join()
    
    pool.close_all()
    print("✓ ConnectionPool tests passed")

def test_transaction_manager():
    db_path = ":memory:"
    pool = ConnectionPool(db_path, pool_size=1)
    tm = TransactionManager(pool)
    
    # Setup table
    conn = pool.acquire()
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
    pool.release(conn)
    
    # Test 1: Basic commit
    with tm as conn:
        conn.execute("INSERT INTO test (val) VALUES (?)", ("hello",))
    
    conn = pool.acquire()
    res = conn.execute("SELECT val FROM test").fetchone()
    pool.release(conn)
    assert res[0] == "hello"
    
    # Test 2: Rollback on exception
    try:
        with tm as conn:
            conn.execute("INSERT INTO test (val) VALUES (?)", ("world",))
            raise ValueError("Force rollback")
    except ValueError:
        pass
        
    conn = pool.acquire()
    res = conn.execute("SELECT COUNT(*) FROM test").fetchone()
    pool.release(conn)
    assert res[0] == 1
    
    pool.close_all()
    print("✓ TransactionManager tests passed")

def test_retry_policy():
    policy = RetryPolicy(max_retries=2, base_delay=0.01)
    
    # Test 1: Success after retry
    attempts = [0]
    def mock_locked():
        if attempts[0] < 1:
            attempts[0] += 1
            raise Exception("database is locked")
        return "OK"
    
    result = policy.execute_with_retry(mock_locked)
    assert result == "OK"
    assert attempts[0] == 1
    
    # Test 2: Max retries exceeded
    def always_locked():
        raise Exception("database is locked")
        
    try:
        policy.execute_with_retry(always_locked)
    except DatabaseLockedError:
        print("✓ RetryPolicy max retries passed")
    else:
        pytest.fail("Should have raised DatabaseLockedError")
        
    print("✓ RetryPolicy tests passed")

if __name__ == "__main__":
    test_connection_pool()
    test_transaction_manager()
    test_retry_policy()
    print("All concurrency unit tests passed!")
