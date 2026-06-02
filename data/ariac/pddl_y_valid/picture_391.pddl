(define (problem picture_391)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery green_battery red_pump green_regulator blue_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear blue_battery)
        (clear green_regulator)
        (clear blue_regulator)
        (on green_battery red_pump)
        (part_at red_pump table)
        (part_at green_regulator regulator_placement)
        (part_at blue_battery battery_placement)
        (part_at blue_regulator table)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
        )
    )
)