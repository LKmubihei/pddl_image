(define (problem picture_336)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
         blue_battery green_regulator red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_pump)
        (clear blue_battery)
        (clear green_regulator)
        (part_at red_pump table)
        (part_at blue_battery battery_placement)
        (part_at green_regulator regulator_placement)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
        )
    )
)