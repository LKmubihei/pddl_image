(define (problem picture_349)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_regulator green_pump blue_battery red_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_regulator)
        (clear blue_battery)
        (clear red_battery)
        (clear green_pump)
        (part_at green_pump table)
        (part_at blue_battery table)
        (part_at red_regulator table)
        (part_at red_battery battery_placement)
    )
    
    (:goal
(and
            (part_at green_pump pump_placement)
            (part_at red_regulator regulator_placement)
        )
    )
)