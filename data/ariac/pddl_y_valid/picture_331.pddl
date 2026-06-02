(define (problem picture_331)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_regulator)
        (part_at green_regulator table)
    )
    
    (:goal
(and
            (part_at green_regulator pump_placement)
        )
    )
)