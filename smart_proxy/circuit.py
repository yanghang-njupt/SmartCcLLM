"""三态熔断状态机 —— 分级冷却 + 指数退避"""


def block_seconds(config, kind):
    """分级冷却时长。kind: quota | server | network"""
    return {
        "quota": config.block_seconds_quota,
        "server": config.block_seconds_server,
        "network": config.block_seconds_network,
    }[kind]


def effective_block(config, kind, fail_count):
    """指数退避：fail_count 越多冷却越久，上限 backoff_max_seconds。"""
    base = block_seconds(config, kind)
    secs = base * (config.backoff_multiplier ** max(0, fail_count - 1))
    return min(secs, config.backoff_max_seconds)
