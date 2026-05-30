(define (problem tv_screw_assembly_case_order_ABCDEFG)
  (:domain tv_screw_assembly)

  (:objects
    power_com - component
    TV_panel - panel
    material_box - box

    ;; 7 screws (A~G) and their corresponding holes
    screw_A screw_B screw_C screw_D screw_E screw_F screw_G - screw
    hole_A  hole_B  hole_C  hole_D  hole_E  hole_F  hole_G  - hole
  )

  (:init
    ;; ===== initial phase =====
    (initial-state)

    ;; ===== resources =====
    (comp-grasp-free)
    (screw-grasp-free)

    ;; ===== materials =====
    (in-material-box power_com material_box)

    ;; ===== screw-hole assignment (静态工艺分配) =====
    (screw-for-hole screw_A hole_A)
    (screw-for-hole screw_B hole_B)
    (screw-for-hole screw_C hole_C)
    (screw-for-hole screw_D hole_D)
    (screw-for-hole screw_E hole_E)
    (screw-for-hole screw_F hole_F)
    (screw-for-hole screw_G hole_G)

    ;; ===== screw availability =====
    (screw-unused screw_A)
    (screw-unused screw_B)
    (screw-unused screw_C)
    (screw-unused screw_D)
    (screw-unused screw_E)
    (screw-unused screw_F)
    (screw-unused screw_G)

    ;; ===== hole states =====
    (hole-empty hole_A)
    (hole-empty hole_B)
    (hole-empty hole_C)
    (hole-empty hole_D)
    (hole-empty hole_E)
    (hole-empty hole_F)
    (hole-empty hole_G)

    ;; ===== process order (固定工序：A -> B -> C -> D -> E -> F -> G) =====
    (requires-predecessor hole_B hole_A)
    (requires-predecessor hole_C hole_B)
    (requires-predecessor hole_D hole_C)
    (requires-predecessor hole_E hole_D)
    (requires-predecessor hole_F hole_E)
    (requires-predecessor hole_G hole_F)
  )

  (:goal
    (and
      ;; 组件必须完成安装
      (comp-on-panel power_com TV_panel)

      ;; 每个螺丝必须拧到对应孔位
      (screw-fastened screw_A hole_A)
      (screw-fastened screw_B hole_B)
      (screw-fastened screw_C hole_C)
      (screw-fastened screw_D hole_D)
      (screw-fastened screw_E hole_E)
      (screw-fastened screw_F hole_F)
      (screw-fastened screw_G hole_G)
    )
  )
)