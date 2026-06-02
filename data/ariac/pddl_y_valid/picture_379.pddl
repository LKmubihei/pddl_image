(define (problem picture_379)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery blue_regulator green_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_battery)
        (clear blue_regulator)
        (on blue_regulator green_pump)
        (part_at green_pump table)
        (part_at red_battery table)
    )
    
    (:goal
(and
            (part_at red_battery battery_placement)
            (part_at green_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)