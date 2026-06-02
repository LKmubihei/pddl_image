(define (problem picture_374)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery red_battery green_regulator green_pump red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_battery)
        (clear green_regulator)
        (clear green_pump)
        (clear red_pump)
        (part_at green_pump table)
        (part_at green_regulator table)
        (part_at red_battery battery_placement)
        (part_at green_battery table)
        (part_at red_pump table)
    )
    
    (:goal
(and
            (part_at green_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)