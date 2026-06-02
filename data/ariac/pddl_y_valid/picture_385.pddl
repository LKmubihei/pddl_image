(define (problem picture_385)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery red_pump green_regulator red_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_battery)
        (on green_regulator red_pump)
        (clear green_regulator)
        (part_at green_battery table)
        (part_at red_pump table)
        (part_at red_battery table)
    )
    
    (:goal
(and
            (part_at green_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)