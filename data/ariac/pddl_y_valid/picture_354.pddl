(define (problem picture_354)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery red_pump green_regulator green_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear green_regulator)
        (clear blue_battery)
        (on green_regulator red_pump)
        (part_at blue_battery table)
        (part_at red_pump table)
        (part_at green_battery table)
    )
    
    (:goal
(and
            (part_at blue_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)