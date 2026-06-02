(define (problem picture_362)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery blue_battery green_pump green_regulator green_regulator_1 blue_battery_1 - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear blue_battery)
        (clear green_pump)
        (clear blue_battery_1)
        (clear green_regulator)
        (clear green_regulator_1)
        (part_at green_pump table)
        (part_at green_battery table)
        (part_at blue_battery_1 table)
        (part_at green_regulator_1 table)
        (part_at green_regulator regulator_placement)
        (part_at blue_battery battery_placement)
    )
    
    (:goal
(and
            (part_at green_pump pump_placement)
        )
    )
)