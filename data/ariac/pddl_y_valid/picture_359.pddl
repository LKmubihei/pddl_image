(define (problem picture_359)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_regulator green_battery red_pump blue_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_regulator)
        (clear green_battery)
        (clear blue_battery)
        (on blue_battery red_pump)
        (part_at blue_regulator table)
        (part_at green_battery table)
        (part_at red_pump table)
    )
    
    (:goal
(and
            (part_at green_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)