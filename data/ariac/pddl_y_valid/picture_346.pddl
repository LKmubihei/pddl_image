(define (problem picture_346)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery green_pump red_regulator green_regulator blue_battery_1 - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear blue_battery_1)
        (clear green_regulator)
        (clear red_regulator)
        (clear green_pump)
        (part_at blue_battery_1 table)
        (part_at green_pump table)
        (part_at green_regulator regulator_placement)
        (part_at blue_battery battery_placement)
        (part_at red_regulator table)
    )
    
    (:goal
(and
            (part_at green_pump pump_placement)
        )
    )
)