(define (problem picture_387)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery blue_battery green_regulator red_pump blue_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_battery)
        (clear blue_battery)
        (clear green_regulator)
        (clear blue_regulator)
        (on blue_regulator red_pump)
        (part_at red_pump table)
        (part_at blue_battery battery_placement)
        (part_at red_battery table)
        (part_at green_regulator regulator_placement)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
        )
    )
)