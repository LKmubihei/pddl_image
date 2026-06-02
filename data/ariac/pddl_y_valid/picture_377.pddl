(define (problem picture_377)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_regulator blue_battery red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (on blue_battery red_pump)
        (clear blue_regulator)
        (clear blue_battery)
        (part_at red_pump table)
        (part_at blue_regulator table)
    )
    
    (:goal
(and
            (part_at blue_battery battery_placement)
            (part_at red_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)