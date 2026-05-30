(define (domain tv_screw_assembly)
  (:requirements
    :typing
    :negative-preconditions
    :disjunctive-preconditions
    :quantified-preconditions
  )

  (:types
    component panel box screw hole
  )

  (:predicates
    ;; ===== phase / progress =====
    (initial-state)
    (power-com-inspected)
    (power-com-placement-done)

    ;; ===== material / resource =====
    (in-material-box ?c - component ?b - box)
    (comp-grasp-free)
    (screw-grasp-free)

    ;; ===== power component states =====
    (comp-in-hand ?c - component)
    (comp-at-panel-area ?c - component ?p - panel)   ; move后到面板区域（粗定位）
    (comp-aligned ?c - component ?p - panel)         ; locating后完成精定位
    (comp-on-panel ?c - component ?p - panel)

    ;; ===== screw process states =====
    ;; fetch_screw() 无参数，因此用“通用抓取状态”表示当前有一颗螺丝被取出
    (screw-fetched)      ; 已取到一颗螺丝，待对孔
    (screw-positioned)   ; 已完成对孔定位，待插入

    ;; 螺丝对象与孔位对象状态
    (screw-unused ?s - screw)
    (screw-for-hole ?s - screw ?h - hole)            ; 静态分配：该螺丝用于该孔位
    (hole-empty ?h - hole)
    (screw-aligned ?s - screw ?h - hole)
    (screw-inserted ?s - screw ?h - hole)
    (screw-fastened ?s - screw ?h - hole)

    ;; ===== order / process constraints =====
    (hole-done ?h - hole)
    (requires-predecessor ?h - hole ?pre - hole)     ; h 的前驱孔位是 pre，必须先完成
  )

  ;; =========================================================
  ;; 组件放置阶段（按你的命名，统一 *_power_com）
  ;; =========================================================

  (:action inspect_power_com
    :parameters ()
    :precondition (and
      (initial-state)
      (not (power-com-inspected))
    )
    :effect (and
      (power-com-inspected)
      (not (initial-state))
    )
  )

  (:action pick_power_com
    :parameters (?c - component ?b - box)
    :precondition (and
      (power-com-inspected)
      (in-material-box ?c ?b)
      (comp-grasp-free)
      (not (comp-in-hand ?c))
    )
    :effect (and
      (comp-in-hand ?c)
      (not (in-material-box ?c ?b))
      (not (comp-grasp-free))
    )
  )

  (:action move_power_com
    :parameters (?c - component ?p - panel)
    :precondition (and
      (comp-in-hand ?c)
      (not (comp-on-panel ?c ?p))
    )
    :effect (and
      (comp-at-panel-area ?c ?p)
      ;; 到了目标区域不代表已精定位
      (not (comp-aligned ?c ?p))
    )
  )

  (:action locating_power_com
    :parameters (?c - component ?p - panel)
    :precondition (and
      (comp-in-hand ?c)
      (comp-at-panel-area ?c ?p)
      (not (comp-aligned ?c ?p))
    )
    :effect (and
      (comp-aligned ?c ?p)
    )
  )

  (:action place_power_com
    :parameters (?c - component ?p - panel)
    :precondition (and
      (comp-in-hand ?c)
      (comp-at-panel-area ?c ?p)
      (comp-aligned ?c ?p)
    )
    :effect (and
      (comp-on-panel ?c ?p)
      (power-com-placement-done)
      (comp-grasp-free)
      (not (comp-in-hand ?c))
      (not (comp-at-panel-area ?c ?p))
    )
  )

  (:action repick_power_com
    :parameters (?c - component ?p - panel)
    :precondition (and
      (comp-on-panel ?c ?p)
      (comp-grasp-free)
      ;; 返工入口：放置后发现不再满足对齐状态时可重抓
      (not (comp-aligned ?c ?p))
    )
    :effect (and
      (comp-in-hand ?c)
      (comp-at-panel-area ?c ?p)
      (not (comp-on-panel ?c ?p))
      (not (power-com-placement-done))
      (not (comp-grasp-free))
    )
  )

  ;; =========================================================
  ;; 螺钉紧固阶段（保持你的原动作名称）
  ;; =========================================================

  (:action fetch_screw
    :parameters ()
    :precondition (and
      (power-com-placement-done)
      (screw-grasp-free)
      (not (screw-fetched))
      (not (screw-positioned))
    )
    :effect (and
      (screw-fetched)
      (not (screw-grasp-free))
    )
  )

  (:action locating_screw
    :parameters (?s - screw ?h - hole)
    :precondition (and
      (power-com-placement-done)
      (screw-fetched)
      (not (screw-positioned))
      (screw-unused ?s)
      (screw-for-hole ?s ?h)
      (hole-empty ?h)
      (not (screw-aligned ?s ?h))

      ;; 工序顺序约束（不依赖PDDL3）：
      ;; 所有前驱孔位必须已完成，当前孔位才能开始该轮螺钉操作
      (forall (?pre - hole)
        (or
          (not (requires-predecessor ?h ?pre))
          (hole-done ?pre)
        )
      )
    )
    :effect (and
      (screw-positioned)
      (screw-aligned ?s ?h)
      (not (screw-fetched))
    )
  )

  (:action insert_screw
    :parameters (?s - screw ?h - hole)
    :precondition (and
      (power-com-placement-done)
      (screw-positioned)
      (screw-aligned ?s ?h)
      (screw-unused ?s)
      (screw-for-hole ?s ?h)
      (hole-empty ?h)
      (not (screw-inserted ?s ?h))
    )
    :effect (and
      (screw-inserted ?s ?h)
      (not (screw-unused ?s))
      (not (hole-empty ?h))
      (not (screw-aligned ?s ?h))
      (not (screw-positioned))
      (screw-grasp-free)
    )
  )

  (:action fasten_screw
    :parameters (?s - screw ?h - hole)
    :precondition (and
      (power-com-placement-done)
      (screw-inserted ?s ?h)
      (not (screw-fastened ?s ?h))
    )
    :effect (and
      (screw-fastened ?s ?h)
      (hole-done ?h)
    )
  )
)