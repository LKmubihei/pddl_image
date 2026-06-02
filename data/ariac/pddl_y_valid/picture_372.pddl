(define (problem picture_372)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery red_pump green_battery red_battery green_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear green_battery)
        (clear red_battery)
        (clear green_regulator)
        (on green_regulator red_pump)
        (part_at red_battery battery_placement)
        (part_at green_battery table)
        (part_at blue_battery table)
        (part_at red_pump table)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)