(define (problem picture_333)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_regulator)
        (part_at blue_regulator table)
    )
    
    (:goal
(and
            (part_at blue_regulator pump_placement)
        )
    )
)