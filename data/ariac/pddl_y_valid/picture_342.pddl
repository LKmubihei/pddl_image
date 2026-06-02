(define (problem picture_342)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_pump blue_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_pump)
        (clear blue_regulator)
        (part_at green_pump table)
        (part_at blue_regulator table)
    )
    
    (:goal
(and
            (part_at green_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)