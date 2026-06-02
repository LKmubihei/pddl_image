(define (problem picture_332)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
         red_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_regulator)
        (part_at red_regulator table)
    )
    
    (:goal
(and
            (part_at red_regulator regulator_placement)
        )
    )
)