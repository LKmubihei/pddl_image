(define (problem picture_316)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump green_regulator blue_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear red_pump)
        (clear green_regulator)
        (clear blue_pump)
        (part_at red_pump table)
        (part_at blue_pump table)
        (part_at green_regulator buffer_placement)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)