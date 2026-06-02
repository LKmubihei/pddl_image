(define (problem picture_357)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery green_pump green_regulator blue_regulator - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear green_regulator)
        (clear blue_regulator)
        (part_at green_battery table)
        (part_at green_pump table)
        (part_at green_regulator table)
        (on blue_regulator green_pump)
    )
    
    (:goal
(and
            (part_at green_battery battery_placement)
            (part_at green_pump pump_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)