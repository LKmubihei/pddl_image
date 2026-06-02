(define (problem picture_315)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
         blue_battery red_regulator red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear red_regulator)
        (clear red_pump)
        (part_at red_pump buffer_placement)
        (part_at blue_battery battery_placement)
        (part_at red_regulator table)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at red_regulator regulator_placement)
        )
    )
)