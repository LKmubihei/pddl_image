(define (problem picture_350)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery red_pump  - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_pump)
        (part_at green_battery table)
        (part_at red_pump pump_placement)
    )
    
    (:goal
(and
            (part_at green_battery battery_placement)
        )
    )
)