(define (problem picture_343)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery blue_regulator green_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear blue_regulator)
        (clear green_pump)
        (part_at blue_battery table)
        (part_at blue_regulator table)
        (part_at green_pump table)
    )
    
    (:goal
(and
            (part_at blue_battery battery_placement)
            (part_at green_pump pump_placement)
            (part_at blue_regulator regulator_placement)
        )
    )
)