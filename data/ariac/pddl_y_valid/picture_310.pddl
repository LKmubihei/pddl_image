(define (problem picture_310)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery green_battery blue_regulator red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear green_battery)
        (clear blue_regulator)
        (clear red_pump)
        (part_at green_battery table)
        (part_at blue_regulator regulator_placement)
        (part_at blue_battery buffer_placement)
        (part_at red_pump pump_placement)
    )
    
    (:goal
(and
            (part_at blue_battery battery_placement)
        )
    )
)