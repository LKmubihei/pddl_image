(define (problem picture_313)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_regulator_1 green_pump green_regulator  - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_pump)
        (clear green_regulator_1)
        (clear green_regulator)
        (part_at green_pump table)
        (part_at green_regulator_1 buffer_placement)
        (part_at green_regulator table)
    )
    
    (:goal
(and
            (part_at green_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)