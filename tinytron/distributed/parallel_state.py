import torch
import torch.distributed as dist

from tinytron.training.config import ParallelConfig

_SEP_GROUP = None
_DP_GROUP = None
_DP_SP_GROUP = None

_SEP_SIZE = None
_DP_SIZE = None
_DP_SP_SIZE = None

def initialize_model_parallel(config: ParallelConfig):
    """
    Initialize the distributed communication grid
    Must be called after dist.init_process_group!
    """
    global _SEP_GROUP, _DP_GROUP, _DP_SP_GROUP, _SEP_SIZE, _DP_SIZE, _DP_SP_SIZE
    sep_size = config.sep_size

    assert dist.is_initialized(), "Must initialize PyTorch distributed environment first!"
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    assert world_size % sep_size == 0, "World size must be divisible by sep_size"
    dp_size = world_size // sep_size
    
    _SEP_SIZE = sep_size
    _DP_SIZE = dp_size
    _DP_SP_SIZE = dp_size * sep_size

    # 1. Build SEP group (continuous Rank)
    for i in range(dp_size):
        sep_ranks = list(range(i * sep_size, (i + 1) * sep_size))
        group = dist.new_group(sep_ranks)
        if rank in sep_ranks:
            _SEP_GROUP = group

    # 2. Build DP group (same offset Rank across SEP groups)
    for j in range(sep_size):
        dp_ranks = [i * sep_size + j for i in range(dp_size)]
        group = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            _DP_GROUP = group

    # 3. Build DP_SP group (same offset Rank across DP groups)
    dp_sp_ranks = list(range(world_size))
    _DP_SP_GROUP = dist.new_group(dp_sp_ranks)


def get_dp_group():
    assert _DP_GROUP is not None, "DP Group is not initialized"
    return _DP_GROUP

def get_dp_world_size():
    return _DP_SIZE

def get_dp_rank():
    return dist.get_rank(group=get_dp_group())

def get_dp_global_rank(group_rank: int = None):
    if group_rank is None:
        group_rank = get_dp_rank()
    return dist.get_global_rank(group=get_dp_group(), group_rank=group_rank)

def _get_dp_group_id_from_layout(global_rank: int, sep_size: int) -> int:
    """
    For the current layout in initialize_model_parallel():
      - DP groups are same offset across SEP groups
      - therefore dp_group_id == global_rank % sep_size
    """
    return global_rank % sep_size

def get_sep_group():
    assert _SEP_GROUP is not None, "SEP Group is not initialized"
    return _SEP_GROUP

def get_sep_world_size():
    return _SEP_SIZE

def get_sep_rank():
    return dist.get_rank(group=get_sep_group())

def get_sep_global_rank(group_rank: int = None):
    if group_rank is None:
        group_rank = get_sep_rank()
    return dist.get_global_rank(group=get_sep_group(), group_rank=group_rank)

def get_sp_group():
    return get_sep_group()

def get_sp_world_size():
    return get_sep_world_size()

def get_sp_rank():
    return get_sep_rank()

def get_sp_global_rank(group_rank: int = None):
    if group_rank is None:
        group_rank = get_sp_rank()
    return dist.get_global_rank(group=get_sp_group(), group_rank=group_rank)

def _get_sep_group_id_from_layout(global_rank: int, sep_size: int) -> int:
    """
    For the current layout in initialize_model_parallel():
      - SEP groups are continuous ranks
      - therefore sep_group_id == global_rank // sep_size
    """
    return global_rank // sep_size

def get_ep_group():
    return get_sep_group()

def get_ep_world_size():
    return get_sep_world_size()

def get_ep_rank():
    return get_sep_rank()

def get_ep_global_rank(group_rank: int = None):
    if group_rank is None:
        group_rank = get_ep_rank()
    return dist.get_global_rank(group=get_ep_group(), group_rank=group_rank)

def get_dp_sp_group():
    assert _DP_SP_GROUP is not None, "DP_SP Group is not initialized"
    return _DP_SP_GROUP

def get_dp_sp_world_size():
    return _DP_SP_SIZE

def get_dp_sp_rank():
    return dist.get_rank(group=get_dp_sp_group())

def get_dp_sp_global_rank(group_rank: int = None):
    if group_rank is None:
        group_rank = get_dp_sp_rank()
    return dist.get_global_rank(group=get_dp_sp_group(), group_rank=group_rank)


def print_model_parallel_topology(
    master_process,
    tag: str = "Parallel Topology",
    group_for_gather=None,
    only_rank0: bool = True,
    flush: bool = True,
):
    """
    Pretty-print topology debug information for current parallel_state layout.

    Requirements:
      - dist.init_process_group(...) has been called
      - initialize_model_parallel(...) has been called

    What it prints:
      1) Static layout table derived from current world_size/sep_size
      2) Static group membership (SEP / DP)
      3) Dynamic per-rank info gathered from all ranks:
         - rank, sp_rank, dp_rank, inferred sep_group_id(data_replica_id)
         - actual sep_members / dp_members from current process view

    Notes:
      - In current layout, `data_replica_id == sep_group_id == global_rank // sep_size`
      - `dp_rank == sep_group_id` (because DP groups are column-wise across SEP rows)
    """
    def _format_rank_list(ranks: list[int]) -> str:
        return "[" + ", ".join(str(r) for r in ranks) + "]"

    assert dist.is_initialized(), "torch.distributed must be initialized before printing topology"
    assert _SEP_GROUP is not None and _DP_GROUP is not None and _DP_SP_GROUP is not None, \
        "Model parallel groups are not initialized. Call initialize_model_parallel() first."

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    sep_size = get_sep_world_size()
    dp_size = get_dp_world_size()

    assert world_size % sep_size == 0
    assert dp_size == world_size // sep_size

    if group_for_gather is None:
        group_for_gather = get_dp_sp_group()

    # ---- local runtime info (dynamic) ----
    sp_rank = get_sp_rank()
    sp_world_size = get_sp_world_size()
    dp_rank = get_dp_rank()
    dp_world_size = get_dp_world_size()

    # inferred IDs under current layout
    sep_group_id = _get_sep_group_id_from_layout(rank, sep_size)   # also data replica id in current design
    dp_group_id = _get_dp_group_id_from_layout(rank, sep_size)

    # actual members as seen from this process
    sep_members = [get_sep_global_rank(i) for i in range(sp_world_size)]
    dp_members = [get_dp_global_rank(i) for i in range(dp_world_size)]

    # pack into a fixed-size tensor for all_gather
    # fields:
    # [0] rank
    # [1] sep_group_id (= data_replica_id)
    # [2] sp_rank
    # [3] sp_world_size
    # [4] dp_group_id
    # [5] dp_rank
    # [6] dp_world_size
    # [7 : 7+sep_size) sep_members
    # [7+sep_size : 7+sep_size+dp_size) dp_members
    payload_len = 7 + sep_size + dp_size
    payload = torch.full(
        (payload_len,),
        fill_value=-1,
        dtype=torch.int64,
        device=torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu"),
    )

    payload[0] = rank
    payload[1] = sep_group_id
    payload[2] = sp_rank
    payload[3] = sp_world_size
    payload[4] = dp_group_id
    payload[5] = dp_rank
    payload[6] = dp_world_size

    for i, r in enumerate(sep_members):
        payload[7 + i] = r
    base = 7 + sep_size
    for i, r in enumerate(dp_members):
        payload[base + i] = r

    gathered = [torch.empty_like(payload) for _ in range(world_size)]
    dist.all_gather(gathered, payload, group=group_for_gather)

    # ---- print only on rank0 (default) ----
    should_print = (rank == 0) if only_rank0 else True
    if not should_print:
        return

    if master_process:
        print(f"\n=== {tag} ===", flush=flush)
        print(
            f"world_size={world_size}, sep_size={sep_size}, dp_size={dp_size} | "
            f"layout: SEP=rows(continuous), DP=cols(same offset)",
            flush=flush,
        )

        # 1) Static topology table
        print("\n[Static Rank Mapping]", flush=flush)
        print(
            f"{'global':>6} | {'sep_gid(data_rep)':>16} | {'sp_rank':>7} | "
            f"{'dp_gid':>6} | {'dp_rank':>7}",
            flush=flush,
        )
        print("-" * 60, flush=flush)
        for r in range(world_size):
            s_gid = _get_sep_group_id_from_layout(r, sep_size)
            s_rank = r % sep_size
            d_gid = s_rank
            d_rank = s_gid
            print(
                f"{r:>6} | {s_gid:>16} | {s_rank:>7} | {d_gid:>6} | {d_rank:>7}",
                flush=flush,
            )

        # 2) Grid view
        print("\n[Grid View] rows=SEP groups(data replicas), cols=SP ranks", flush=flush)
        header = "        " + " ".join([f"sp{j:>4}" for j in range(sep_size)])
        print(header, flush=flush)
        print("        " + "------" * sep_size, flush=flush)
        for sep_gid in range(dp_size):
            row = [sep_gid * sep_size + sp for sp in range(sep_size)]
            cells = " ".join([f"{r:>5}" for r in row])
            print(f"SEP[{sep_gid:>2}] | {cells}", flush=flush)

        # 3) Static group membership
        print("\n[Static Group Membership]", flush=flush)
        print("SEP groups:", flush=flush)
        for i in range(dp_size):
            sep_ranks = list(range(i * sep_size, (i + 1) * sep_size))
            print(f"  SEP[{i}] = {_format_rank_list(sep_ranks)}", flush=flush)

        print("DP groups:", flush=flush)
        for j in range(sep_size):
            dp_ranks = [i * sep_size + j for i in range(dp_size)]
            print(f"  DP[{j}]  = {_format_rank_list(dp_ranks)}", flush=flush)

        # 4) Dynamic per-rank info (real process-group view)
        print("\n[Dynamic Per-Rank View (gathered)]", flush=flush)
        print(
            f"{'rank':>4} | {'sep_gid':>7} | {'sp_r':>4}/{ 'sp_ws':<4} | "
            f"{'dp_gid':>6} | {'dp_r':>4}/{ 'dp_ws':<4} | {'sep_members':>16} | {'dp_members':>16}",
            flush=flush,
        )
        print("-" * 140, flush=flush)

        rows = []
        for t in gathered:
            vals = [int(x) for x in t.cpu().tolist()]
            r = vals[0]
            s_gid = vals[1]
            s_r = vals[2]
            s_ws = vals[3]
            d_gid = vals[4]
            d_r = vals[5]
            d_ws = vals[6]
            s_members = vals[7 : 7 + sep_size]
            d_members = vals[7 + sep_size : 7 + sep_size + dp_size]
            rows.append((r, s_gid, s_r, s_ws, d_gid, d_r, d_ws, s_members, d_members))

        rows.sort(key=lambda x: x[0])

        for (r, s_gid, s_r, s_ws, d_gid, d_r, d_ws, s_members, d_members) in rows:
            print(
                f"{r:>4} | {s_gid:>7} | {s_r:>4}/{s_ws:<4} | "
                f"{d_gid:>6} | {d_r:>4}/{d_ws:<4} | "
                f"{str(s_members):>16} | {str(d_members):>16}",
                flush=flush,
            )

        print("\n[Interpretation]", flush=flush)
        print(
            "- Each SEP group (row) is one data replica: ranks in the same SEP group should receive the SAME batch "
            "before sequence slicing by sp_rank.",
            flush=flush,
        )
        print(
            "- Each DP group (column) synchronizes dense (non-expert) gradients across replicas for the same sp offset.",
            flush=flush,
        )
        print(
            "- In this layout, sampler replica id can be taken as sep_group_id, which is numerically equal to dp_rank.",
            flush=flush,
        )
        print("=" * (len(tag) + 8), flush=flush)