(define (problem picture_380)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery blue_regulator red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_battery)
        (clear blue_regulator)
        (part_at red_battery table)
        (part_at red_pump table)
        (on blue_regulator red_pump)
    )
    
    (:goal
(and
            (part_at red_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)