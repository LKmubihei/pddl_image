(define (problem picture_398)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery red_battery_1 blue_regulator red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_battery)
        (clear red_battery_1)
        (clear blue_regulator)
        (on blue_regulator red_pump)
        (part_at red_pump table)
        (part_at red_battery battery_placement)
        (part_at red_battery_1 table)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)