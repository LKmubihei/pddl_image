(define (problem picture_386)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery green_battery red_pump green_regulator  - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_battery)
        (clear green_regulator)
        (part_at red_battery battery_placement)
        (part_at green_battery table)
        (on green_regulator red_pump)
        (part_at red_pump table)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)