(define (problem picture_362)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery blue_battery green_pump green_regulator red_pump - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear blue_battery)
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
            (part_at green_pump pump_placement)
        )
    )
)