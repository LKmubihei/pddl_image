(define (problem picture_368)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery green_battery red_pump_1 green_regulator red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_battery)
        (clear green_battery)
        (clear green_regulator)
        (clear red_pump)
        (on green_regulator red_pump_1)
        (part_at red_pump_1 table)
        (part_at green_battery table)
        (part_at red_battery table)
        (part_at red_pump pump_placement)
    )
    
    (:goal
(and
            (part_at red_battery battery_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)