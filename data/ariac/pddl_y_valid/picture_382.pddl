(define (problem picture_382)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump blue_regulator green_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_regulator)
        (clear red_pump)
        (part_at red_pump table)
        (part_at green_battery table)
        (on blue_regulator green_battery)
    )
    
    (:goal
(and
            (part_at green_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)