(define (problem picture_355)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery red_pump green_regulator blue_battery blue_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear green_regulator)
        (clear blue_regulator)
        (clear blue_battery)
        (on blue_battery red_pump)
        (part_at green_battery table)
        (part_at red_pump table)
        (part_at blue_regulator table)
        (part_at green_regulator table)
    )
    
    (:goal
(and
            (part_at green_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)