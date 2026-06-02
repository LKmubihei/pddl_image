(define (problem picture_304)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery green_pump blue_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_battery)
        (clear green_pump)
        (clear blue_regulator)
        (part_at blue_regulator table)
        (part_at green_pump table)
        (part_at red_battery buffer_placement)
    )
    
    (:goal
(and
            (part_at red_battery battery_placement)
            (part_at green_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)