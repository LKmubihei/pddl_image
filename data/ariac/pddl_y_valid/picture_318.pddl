(define (problem picture_318)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_pump green_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_pump)
        (clear green_regulator)
        (part_at blue_pump table)
        (part_at green_regulator buffer_placement)
    )
    
    (:goal
(and
            (part_at green_regulator regulator_placement)
        )
    )
)