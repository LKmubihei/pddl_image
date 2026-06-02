(define (problem picture_376)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump blue_regulator green_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (on green_battery red_pump)
        (clear blue_regulator)
        (clear green_battery)
        (part_at red_pump table)
        (part_at blue_regulator table)
    )
    
    (:goal
(and
            (part_at green_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)