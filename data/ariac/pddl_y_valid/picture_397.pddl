(define (problem picture_397)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery blue_battery red_pump green_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_battery)
        (clear blue_battery)
        (on green_regulator red_pump)
        (clear green_regulator)
        (part_at red_pump table)
        (part_at blue_battery table)
        (part_at red_battery battery_placement)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)