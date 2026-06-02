(define (problem picture_396)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_regulator blue_battery green_pump red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear blue_regulator)
        (clear red_pump)
        (on blue_battery green_pump)
        (part_at blue_regulator table)
        (part_at green_pump table)
        (part_at red_pump pump_placement)
    )
    
    (:goal
(and
            (part_at blue_battery battery_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)