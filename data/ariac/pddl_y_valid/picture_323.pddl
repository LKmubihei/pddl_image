(define (problem picture_323)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery green_pump green_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_pump)
        (clear blue_battery)
        (clear green_regulator)
        (part_at green_pump table)
        (part_at green_regulator table)
        (part_at blue_battery buffer_placement)
    )
    
    (:goal
(and
            (part_at green_pump pump_placement)
            (part_at green_regulator regulator_placement)
            (part_at blue_battery battery_placement)
        )
    )
)