(define (problem picture_368)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery green_battery green_pump green_regulator red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear green_battery)
        (clear green_pump)
        (clear green_regulator)
        (clear red_pump)
        (part_at green_pump table)
        (part_at green_battery table)
        (part_at green_regulator table)
        (part_at blue_battery table)
        (part_at red_pump pump_placement)
    )
    
    (:goal
(and
            (part_at blue_battery battery_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)