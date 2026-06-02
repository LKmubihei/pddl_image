(define (problem picture_303)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
         red_pump green_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_pump)
        (clear green_regulator)
        (part_at red_pump table)
        (part_at green_regulator buffer_placement)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)