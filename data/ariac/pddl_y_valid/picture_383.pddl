(define (problem picture_383)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery red_pump green_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery)
        (on green_regulator red_pump)
        (clear green_regulator)
        (part_at blue_battery table)
        (part_at red_pump table)
    )
    
    (:goal
(and
            (part_at blue_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)