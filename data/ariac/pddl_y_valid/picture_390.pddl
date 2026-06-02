(define (problem picture_390)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery red_battery blue_battery green_regulator red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_battery)
        (clear blue_battery)
        (on blue_battery red_pump)
        (clear green_regulator)
        (part_at green_battery table)
        (part_at red_battery table)
        (part_at green_regulator table)
        (part_at red_pump table)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)