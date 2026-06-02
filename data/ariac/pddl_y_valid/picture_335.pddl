(define (problem picture_335)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_pump)
        (part_at red_pump table)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
        )
    )
)