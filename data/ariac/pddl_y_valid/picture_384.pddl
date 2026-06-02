(define (problem picture_384)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery green_regulator red_pump blue_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear red_battery)
        (on green_regulator red_pump)
        (clear green_regulator)
        (part_at blue_battery table)
        (part_at red_pump table)
        (part_at red_battery table)
    )
    
    (:goal
(and
            (part_at red_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)