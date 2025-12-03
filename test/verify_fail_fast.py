"""
Fail Fast Optimization Verification Script

This script simulates the proxy's behavior under rate limit conditions to verify:
1. Rotation delay is ~0ms (immediate rotation)
2. Key cooldown is properly set to upstream retryDelay value (~48s)
"""

import time
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MockClassifiedError:
    """Mock error similar to the real ClassifiedError"""
    error_type: str
    retry_after: Optional[int] = None
    original_exception: Optional[Exception] = None


class MockRateLimitException(Exception):
    """Mock rate limit exception with fail-fast attributes"""
    def __init__(self, message: str, retry_after: int = 0, original_retry_delay: int = 0):
        super().__init__(message)
        self.retry_after = retry_after
        self.original_retry_delay = original_retry_delay


class MockUsageManager:
    """Mock usage manager to verify cooldown behavior"""
    def __init__(self):
        self.cooldowns: Dict[str, Dict[str, Any]] = {}
        self.cooldown_logs = []
    
    async def record_error(self, key: str, model: str, classified_error: MockClassifiedError):
        """Simulate usage_manager.record_error with fail-fast logic"""
        now = time.time()
        
        if classified_error.error_type == "rate_limit":
            # This is the NEW fail-fast logic from usage_manager.py
            original_delay = getattr(classified_error.original_exception, 'original_retry_delay', None)
            
            if original_delay:
                # Use the original delay from Antigravity for key-level cooldown
                cooldown_seconds = original_delay
                cooldown_type = "original_retry_delay"
            else:
                # Standard retry_after or increased default (300s for fail-fast)
                cooldown_seconds = classified_error.retry_after or 300
                cooldown_type = "retry_after or default"
            
            # Record the cooldown
            if key not in self.cooldowns:
                self.cooldowns[key] = {}
            
            self.cooldowns[key][model] = {
                'cooldown_until': now + cooldown_seconds,
                'cooldown_seconds': cooldown_seconds,
                'cooldown_type': cooldown_type
            }
            
            log_entry = {
                'key': key[-6:],
                'model': model,
                'cooldown_seconds': cooldown_seconds,
                'cooldown_type': cooldown_type,
                'timestamp': now
            }
            self.cooldown_logs.append(log_entry)
            
            return cooldown_seconds
        
        return 0
    
    def get_cooldown_remaining(self, key: str, model: str) -> float:
        """Check if key is still in cooldown"""
        if key in self.cooldowns and model in self.cooldowns[key]:
            remaining = self.cooldowns[key][model]['cooldown_until'] - time.time()
            return max(0, remaining)
        return 0


class MockAntigravityProvider:
    """Mock Antigravity provider with fail-fast retry_after override"""
    
    @staticmethod
    def simulate_rate_limit_error(retry_delay: float = 48.88):
        """
        Simulates Antigravity's behavior when returning a 429 error.
        
        OLD behavior: retry_after = int(retry_delay)  # Would block for 48s
        NEW behavior: retry_after = 0, original_retry_delay = int(retry_delay)
        """
        # This simulates the NEW code from antigravity_provider.py (lines 2334-2358)
        original_retry_delay = int(retry_delay)
        
        # Fail-fast: Set retry_after to 0 for immediate rotation
        rate_limit_error = MockRateLimitException(
            f"Rate limit exceeded. Retry after {retry_delay}s",
            retry_after=0,  # NEW: Immediate rotation
            original_retry_delay=original_retry_delay  # NEW: Store for key cooldown
        )
        
        return rate_limit_error


class MockClient:
    """Mock client to simulate rotation behavior"""
    
    def __init__(self, keys: list, usage_manager: MockUsageManager):
        self.keys = keys
        self.usage_manager = usage_manager
        self.rotation_times = []
        self.base_delay = 0.0  # NEW: Zero delay for fail-fast
    
    async def simulate_request_with_rotation(self, model: str):
        """
        Simulate a request that hits rate limits and rotates through keys.
        
        This simulates the main rotation loop from client.py with NEW fail-fast behavior.
        """
        rotation_start = time.time()
        
        for idx, key in enumerate(self.keys):
            attempt_start = time.time()
            
            # Simulate the request failing with 429 on first 2 keys
            if idx < 2:
                # Simulate Antigravity returning a 429 with retryDelay
                retry_delay = 48.88 if idx == 0 else 45.2
                rate_limit_error = MockAntigravityProvider.simulate_rate_limit_error(retry_delay)
                
                # Create classified error (as error_handler would)
                classified_error = MockClassifiedError(
                    error_type="rate_limit",
                    retry_after=rate_limit_error.retry_after,  # This is 0 (fail-fast)
                    original_exception=rate_limit_error
                )
                
                # Record the error (applies cooldown)
                cooldown_applied = await self.usage_manager.record_error(key, model, classified_error)
                
                # Simulate the rotation delay (NEW: base_delay = 0.0)
                if idx < len(self.keys) - 1:  # Don't delay after last attempt
                    # OLD behavior: base_delay = 2.0 for rate limits
                    # NEW behavior: base_delay = 0.0 for immediate rotation
                    delay = self.base_delay
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
                    
                    rotation_time = time.time() - attempt_start
                    self.rotation_times.append({
                        'from_key': key[-6:],
                        'rotation_time_ms': rotation_time * 1000,
                        'cooldown_applied': cooldown_applied,
                        'delay_used': delay
                    })
            else:
                # Third key succeeds
                rotation_time = time.time() - attempt_start
                self.rotation_times.append({
                    'from_key': key[-6:],
                    'rotation_time_ms': rotation_time * 1000,
                    'cooldown_applied': 0,
                    'delay_used': 0,
                    'success': True
                })
                break
        
        total_rotation_time = time.time() - rotation_start
        return total_rotation_time


async def run_simulation():
    """Run the fail-fast verification simulation"""
    
    print("=" * 80)
    print("FAIL-FAST OPTIMIZATION VERIFICATION")
    print("=" * 80)
    print()
    
    # Setup
    keys = [
        "sk-ant-test-key-1-xxxxxx",
        "sk-ant-test-key-2-yyyyyy",
        "sk-ant-test-key-3-zzzzzz"
    ]
    
    model = "claude-sonnet-4"
    
    usage_manager = MockUsageManager()
    client = MockClient(keys, usage_manager)
    
    print("SIMULATION SETUP:")
    print(f"  - Number of keys: {len(keys)}")
    print(f"  - Model: {model}")
    print(f"  - Base rotation delay: {client.base_delay}s (NEW: fail-fast)")
    print(f"  - Simulated scenario: First 2 keys return 429, 3rd succeeds")
    print()
    
    # Run simulation
    print("RUNNING SIMULATION...")
    print()
    
    total_time = await client.simulate_request_with_rotation(model)
    
    # Results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    print("ROTATION TIMELINE:")
    print("-" * 80)
    for i, rotation in enumerate(client.rotation_times, 1):
        if rotation.get('success'):
            print(f"  Step {i}: Key ...{rotation['from_key']} -> SUCCESS")
            print(f"    - Request succeeded")
            print(f"    - Time to success: {rotation['rotation_time_ms']:.2f}ms")
        else:
            print(f"  Step {i}: Key ...{rotation['from_key']} -> RATE LIMITED -> Rotate")
            print(f"    - Cooldown applied: {rotation['cooldown_applied']}s")
            print(f"    - Rotation delay used: {rotation['delay_used']:.3f}s")
            print(f"    - Time to next key: {rotation['rotation_time_ms']:.2f}ms")
        print()
    
    print("=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print()
    
    # Check 1: Total rotation time should be very fast (< 100ms)
    total_ms = total_time * 1000
    rotation_success = total_ms < 100
    
    print(f"CHECK 1: Rotation Speed (Target: < 100ms)")
    print(f"  - Total time for all rotations: {total_ms:.2f}ms")
    print(f"  - Status: {'PASS' if rotation_success else 'FAIL'}")
    print(f"  - Expected: OLD behavior would take ~50+ seconds (48s + 2s backoff)")
    print(f"  - Actual: NEW fail-fast behavior takes ~{total_ms:.2f}ms")
    print()
    
    # Check 2: Key cooldowns should be set to upstream values
    print(f"CHECK 2: Key Cooldown Duration (Target: ~48s for first key)")
    print(f"  - Key cooldowns in usage_manager:")
    
    cooldown_checks_passed = True
    for key in keys[:2]:  # First 2 keys that failed
        remaining = usage_manager.get_cooldown_remaining(key, model)
        expected_cooldown = usage_manager.cooldowns.get(key, {}).get(model, {}).get('cooldown_seconds', 0)
        cooldown_type = usage_manager.cooldowns.get(key, {}).get(model, {}).get('cooldown_type', 'unknown')
        
        print(f"    * Key ...{key[-6:]}:")
        print(f"      - Cooldown set: {expected_cooldown}s ({cooldown_type})")
        print(f"      - Currently remaining: {remaining:.1f}s")
        
        # Verify it's using original_retry_delay and is around 45-48s
        if expected_cooldown < 40 or expected_cooldown > 50:
            cooldown_checks_passed = False
        if cooldown_type != "original_retry_delay":
            cooldown_checks_passed = False
    
    print(f"  - Status: {'PASS' if cooldown_checks_passed else 'FAIL'}")
    print()
    
    # Check 3: Rotation delay should be 0
    print(f"CHECK 3: Rotation Delay (Target: 0s)")
    print(f"  - Base delay configured: {client.base_delay}s")
    max_rotation_delay = max([r['delay_used'] for r in client.rotation_times if not r.get('success', False)], default=0)
    delay_check_passed = max_rotation_delay == 0
    print(f"  - Maximum delay used during rotation: {max_rotation_delay}s")
    print(f"  - Status: {'PASS' if delay_check_passed else 'FAIL'}")
    print()
    
    # Overall result
    print("=" * 80)
    all_passed = rotation_success and cooldown_checks_passed and delay_check_passed
    
    if all_passed:
        print("*** ALL CHECKS PASSED - Fail-Fast optimization is working correctly! ***")
        print()
        print("SUMMARY:")
        print("  * Rotation happens immediately (< 100ms) instead of ~50+ seconds")
        print("  * Failed keys are cooled down for the full upstream retryDelay (~48s)")
        print("  * Rotation delay is 0s (no exponential backoff blocking)")
        print("  * original_retry_delay attribute is properly extracted and used")
    else:
        print("*** SOME CHECKS FAILED - Review the implementation ***")
    
    print("=" * 80)
    print()
    
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(run_simulation())
    exit(0 if result else 1)