# --------------- 0) 导入与静音（可选） ---------------
import os
import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader

# 关闭引擎 credits 的打印（不影响功能）
up.shortcuts.get_environment().credits_stream = None

# --------------- 1) 从文件加载 PDDL：domain + problem ---------------
_PLAN_DIR = os.path.dirname(os.path.abspath(__file__))
domain_file = os.path.join(_PLAN_DIR, "domain.pddl")
problem_file = os.path.join(_PLAN_DIR, "p_real.pddl")

reader = PDDLReader()
problem = reader.parse_problem(domain_file, problem_file)  # ← 从文件解析

# --------------- 2) 求解：自动选引擎 ---------------
with OneshotPlanner(name="fast-downward",problem_kind=problem.kind) as planner:
    result = planner.solve(problem)
    if result.status in up.engines.results.POSITIVE_OUTCOMES:
        print(f"使用引擎 {planner.name} 得到计划：")
        print(result.plan)  # 顺序计划，每行一个动作
    else:
        raise RuntimeError(f"求解失败：{result.status}")

# # --------------- 3) 验证：形式化校验计划 ---------------
# plan = result.plan
# with PlanValidator(problem_kind=problem.kind, plan_kind=plan.kind) as validator:
#     vres = validator.validate(problem, plan)
#     print("验证结果：", vres)  # 示例输出：status: VALID, engine: Tamer
#     # 更严格的检查：
#     if vres.status.name != "VALID":
#         raise RuntimeError("计划未通过验证")
