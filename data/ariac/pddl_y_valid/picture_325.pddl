(define (problem picture_325)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery green_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear green_regulator)
        (part_at blue_battery table)
        (part_at green_regulator buffer_placement)
    )
    
    (:goal
(and
            (part_at blue_battery battery_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)